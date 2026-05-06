// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "./ShareToken.sol";
import "./NAVOracle.sol";

/**
 * @title SentinelFund
 * @notice Tokenized hedge fund vault.
 *
 *  Architecture:
 *  ┌─────────────────────────────────────────────────────┐
 *  │  Investor deposits USDC → receives SNTL shares      │
 *  │  Share price = NAV / totalSupply                    │
 *  │  Fund manager (trading system) operates capital     │
 *  │  Investors withdraw USDC proportional to shares     │
 *  └─────────────────────────────────────────────────────┘
 *
 *  Safety mechanisms:
 *  - Max drawdown kill-switch: halts withdrawals if DD > 10%
 *  - Cooldown period: 24h between deposit and withdrawal
 *  - Management fee: 2% annually (charged on withdrawal)
 *  - Performance fee: 20% of profits above high-water mark
 *  - Max single deposit: $500,000 USDC (whale protection)
 */
contract SentinelFund is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    // ── Constants ─────────────────────────────────────────────────────────────
    uint256 public constant PRECISION          = 1e18;
    uint256 public constant INITIAL_SHARE_PRICE = 1e18;    // $1.00
    uint256 public constant WITHDRAWAL_COOLDOWN = 24 hours;
    uint256 public constant MAX_DEPOSIT_USDC   = 500_000e6; // $500k
    uint256 public constant MGMT_FEE_BPS       = 200;       // 2% annual
    uint256 public constant PERF_FEE_BPS       = 2000;      // 20%
    uint256 public constant MAX_DRAWDOWN_BPS   = 1000;      // 10% halt threshold
    uint256 public constant YEAR_SECONDS       = 365 days;

    // ── State ─────────────────────────────────────────────────────────────────
    IERC20      public immutable usdc;
    ShareToken  public immutable shares;
    NAVOracle   public immutable oracle;

    bool    public  killSwitchActive;
    bool    public  depositsHalted;
    uint256 public  highWaterMarkPerShare;  // NAV high-water mark for perf fees
    uint256 public  totalDeposited;
    uint256 public  totalWithdrawn;

    mapping(address => uint256) public depositTimestamp;
    mapping(address => uint256) public costBasis;   // avg USDC per share

    address public feeRecipient;

    // ── Events ────────────────────────────────────────────────────────────────
    event Deposited(address indexed investor, uint256 usdc_amount, uint256 shares_minted, uint256 share_price);
    event Withdrawn(address indexed investor, uint256 shares_burned, uint256 usdc_returned, uint256 fees_charged);
    event KillSwitchTripped(uint256 drawdown_bps, uint256 timestamp);
    event KillSwitchReset(address indexed operator);
    event FeesCharged(address indexed investor, uint256 mgmt_fee, uint256 perf_fee);

    // ── Errors ─────────────────────────────────────────────────────────────────
    error KillSwitchActive();
    error DepositsHalted();
    error CooldownNotElapsed(uint256 unlockAt);
    error ExceedsMaxDeposit(uint256 amount, uint256 max);
    error InsufficientShares(uint256 requested, uint256 balance);
    error StaleNAV();
    error ZeroAmount();

    constructor(
        address _usdc,
        address _shares,
        address _oracle,
        address _feeRecipient
    ) Ownable(msg.sender) {
        usdc          = IERC20(_usdc);
        shares        = ShareToken(_shares);
        oracle        = NAVOracle(_oracle);
        feeRecipient  = _feeRecipient;
        highWaterMarkPerShare = INITIAL_SHARE_PRICE;
    }

    // ── Deposit ───────────────────────────────────────────────────────────────
    /**
     * @notice Deposit USDC and receive SNTL fund shares.
     * @param usdc_amount Amount of USDC to deposit (6 decimals).
     */
    function deposit(uint256 usdc_amount) external nonReentrant {
        if (killSwitchActive) revert KillSwitchActive();
        if (depositsHalted)   revert DepositsHalted();
        if (usdc_amount == 0) revert ZeroAmount();
        if (usdc_amount > MAX_DEPOSIT_USDC) revert ExceedsMaxDeposit(usdc_amount, MAX_DEPOSIT_USDC);

        uint256 nav = _currentNAVPerShare();
        // shares_minted = usdc_amount * PRECISION / nav_per_share
        // usdc has 6 decimals, nav has 18 → normalize
        uint256 usdc_normalized = usdc_amount * 1e12;  // 6 → 18 decimals
        uint256 shares_minted   = usdc_normalized * PRECISION / nav;

        usdc.safeTransferFrom(msg.sender, address(this), usdc_amount);
        shares.mint(msg.sender, shares_minted);

        depositTimestamp[msg.sender] = block.timestamp;
        totalDeposited += usdc_amount;

        emit Deposited(msg.sender, usdc_amount, shares_minted, nav);
    }

    // ── Withdraw ──────────────────────────────────────────────────────────────
    /**
     * @notice Burn SNTL shares and receive USDC proportional to current NAV.
     * @param share_amount Number of shares to redeem.
     */
    function withdraw(uint256 share_amount) external nonReentrant {
        if (killSwitchActive) revert KillSwitchActive();
        if (share_amount == 0) revert ZeroAmount();

        uint256 balance = shares.balanceOf(msg.sender);
        if (balance < share_amount) revert InsufficientShares(share_amount, balance);

        uint256 unlock = depositTimestamp[msg.sender] + WITHDRAWAL_COOLDOWN;
        if (block.timestamp < unlock) revert CooldownNotElapsed(unlock);

        uint256 nav = _currentNAVPerShare();
        // usdc_due = share_amount * nav / PRECISION (result in 18 decimals → convert to 6)
        uint256 gross_usdc_18 = share_amount * nav / PRECISION;
        uint256 gross_usdc    = gross_usdc_18 / 1e12;  // 18 → 6 decimals

        // Calculate fees
        (uint256 mgmt_fee, uint256 perf_fee) = _calculateFees(
            msg.sender, share_amount, nav, gross_usdc
        );
        uint256 net_usdc = gross_usdc - mgmt_fee - perf_fee;

        shares.burn(msg.sender, share_amount);
        usdc.safeTransfer(msg.sender, net_usdc);
        if (mgmt_fee + perf_fee > 0) {
            usdc.safeTransfer(feeRecipient, mgmt_fee + perf_fee);
            emit FeesCharged(msg.sender, mgmt_fee, perf_fee);
        }

        totalWithdrawn += gross_usdc;
        emit Withdrawn(msg.sender, share_amount, net_usdc, mgmt_fee + perf_fee);
    }

    // ── Kill switch ───────────────────────────────────────────────────────────
    /**
     * @notice Called by the oracle contract when drawdown exceeds MAX_DRAWDOWN_BPS.
     */
    function checkAndTripKillSwitch() external {
        NAVOracle.NAVReport memory nav = oracle.getLatestNAV();
        if (nav.max_drawdown_bps >= MAX_DRAWDOWN_BPS && !killSwitchActive) {
            killSwitchActive = true;
            emit KillSwitchTripped(nav.max_drawdown_bps, block.timestamp);
        }
    }

    function resetKillSwitch() external onlyOwner {
        killSwitchActive = false;
        emit KillSwitchReset(msg.sender);
    }

    function haltDeposits(bool halt) external onlyOwner {
        depositsHalted = halt;
    }

    // ── View functions ────────────────────────────────────────────────────────
    function currentNAVPerShare() external view returns (uint256) {
        return _currentNAVPerShare();
    }

    function totalAUM() external view returns (uint256) {
        return usdc.balanceOf(address(this));
    }

    function shareValue(address investor) external view returns (uint256 usdc_value) {
        uint256 bal = shares.balanceOf(investor);
        uint256 nav = _currentNAVPerShare();
        return bal * nav / PRECISION / 1e12;
    }

    // ── Internal ──────────────────────────────────────────────────────────────
    function _currentNAVPerShare() internal view returns (uint256) {
        if (oracle.isStale()) revert StaleNAV();
        return oracle.getLatestNAV().nav_per_share;
    }

    function _calculateFees(
        address investor,
        uint256 share_amount,
        uint256 nav,
        uint256 gross_usdc
    ) internal returns (uint256 mgmt_fee, uint256 perf_fee) {
        // Management fee: 2% annual, prorated by time held
        uint256 held_seconds = block.timestamp - depositTimestamp[investor];
        mgmt_fee = (gross_usdc * MGMT_FEE_BPS * held_seconds) / (10_000 * YEAR_SECONDS);

        // Performance fee: 20% of profit above high-water mark
        if (nav > highWaterMarkPerShare) {
            uint256 profit_per_share = nav - highWaterMarkPerShare;
            uint256 total_profit_usdc = (share_amount * profit_per_share / PRECISION) / 1e12;
            perf_fee = (total_profit_usdc * PERF_FEE_BPS) / 10_000;
            highWaterMarkPerShare = nav;
        }
    }
}
