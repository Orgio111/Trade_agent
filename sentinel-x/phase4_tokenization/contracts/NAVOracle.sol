// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/access/AccessControl.sol";

/**
 * @title NAVOracle
 * @notice Off-chain → on-chain bridge for fund performance data.
 *         Authorized reporters (the trading system) push NAV + PnL.
 *         Uses a 2-of-3 multi-reporter threshold to prevent manipulation.
 */
contract NAVOracle is AccessControl {
    bytes32 public constant REPORTER_ROLE = keccak256("REPORTER_ROLE");

    uint256 public constant PRECISION = 1e18;
    uint256 public constant STALENESS_PERIOD = 1 hours;
    uint256 public constant MIN_REPORTERS = 2;   // 2-of-3 consensus

    struct NAVReport {
        uint256 nav_per_share;       // in PRECISION units (1e18 = $1.00)
        uint256 total_aum_usd;       // total AUM in PRECISION
        int256  daily_pnl_bps;       // basis points (signed)
        uint256 sharpe_ratio;        // × 100 (e.g. 150 = 1.50 Sharpe)
        uint256 max_drawdown_bps;    // basis points
        uint256 timestamp;
    }

    // Reporter → latest submission
    mapping(address => NAVReport) private _pendingReports;
    address[] private _reporters;

    // Finalized NAV (requires MIN_REPORTERS consensus)
    NAVReport public finalizedNAV;
    uint256 public lastFinalizedAt;

    event NAVSubmitted(address indexed reporter, uint256 nav_per_share, uint256 timestamp);
    event NAVFinalized(uint256 nav_per_share, uint256 total_aum, int256 daily_pnl_bps);

    constructor(address[] memory reporters) {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        for (uint i = 0; i < reporters.length; i++) {
            _grantRole(REPORTER_ROLE, reporters[i]);
            _reporters.push(reporters[i]);
        }
    }

    /**
     * @notice Submit a NAV report. Automatically finalizes if consensus reached.
     */
    function submitNAV(
        uint256 nav_per_share,
        uint256 total_aum_usd,
        int256  daily_pnl_bps,
        uint256 sharpe_ratio,
        uint256 max_drawdown_bps
    ) external onlyRole(REPORTER_ROLE) {
        _pendingReports[msg.sender] = NAVReport({
            nav_per_share:    nav_per_share,
            total_aum_usd:    total_aum_usd,
            daily_pnl_bps:    daily_pnl_bps,
            sharpe_ratio:     sharpe_ratio,
            max_drawdown_bps: max_drawdown_bps,
            timestamp:        block.timestamp
        });

        emit NAVSubmitted(msg.sender, nav_per_share, block.timestamp);
        _tryFinalize();
    }

    /**
     * @notice Attempt to finalize NAV if MIN_REPORTERS agree within 5% tolerance.
     */
    function _tryFinalize() internal {
        uint256 agreementCount = 0;
        uint256 sumNav = 0;
        uint256 baseNav = _pendingReports[_reporters[0]].nav_per_share;

        for (uint i = 0; i < _reporters.length; i++) {
            NAVReport memory r = _pendingReports[_reporters[i]];
            if (r.timestamp == 0) continue;
            if (block.timestamp - r.timestamp > STALENESS_PERIOD) continue;

            // 5% tolerance band
            uint256 diff = r.nav_per_share > baseNav
                ? r.nav_per_share - baseNav
                : baseNav - r.nav_per_share;
            if (diff * 100 <= baseNav * 5) {
                agreementCount++;
                sumNav += r.nav_per_share;
            }
        }

        if (agreementCount >= MIN_REPORTERS) {
            // Use median (average of agreeing reporters)
            NAVReport memory ref = _pendingReports[_reporters[0]];
            finalizedNAV = NAVReport({
                nav_per_share:    sumNav / agreementCount,
                total_aum_usd:    ref.total_aum_usd,
                daily_pnl_bps:    ref.daily_pnl_bps,
                sharpe_ratio:     ref.sharpe_ratio,
                max_drawdown_bps: ref.max_drawdown_bps,
                timestamp:        block.timestamp
            });
            lastFinalizedAt = block.timestamp;
            emit NAVFinalized(finalizedNAV.nav_per_share, finalizedNAV.total_aum_usd, finalizedNAV.daily_pnl_bps);
        }
    }

    function getLatestNAV() external view returns (NAVReport memory) {
        require(lastFinalizedAt > 0, "NAVOracle: no finalized NAV yet");
        require(block.timestamp - lastFinalizedAt <= STALENESS_PERIOD, "NAVOracle: NAV is stale");
        return finalizedNAV;
    }

    function isStale() external view returns (bool) {
        return block.timestamp - lastFinalizedAt > STALENESS_PERIOD;
    }
}
