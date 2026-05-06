// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title SentinelShare
 * @notice ERC-20 share token representing pro-rata ownership of the Sentinel-X fund.
 *         Only the SentinelFund contract can mint/burn shares.
 *         1 share = 1 unit of NAV at issuance time.
 */
contract SentinelShare is ERC20, Ownable {
    address public fund;

    modifier onlyFund() {
        require(msg.sender == fund, "SentinelShare: caller is not the fund");
        _;
    }

    constructor(address _fund) ERC20("Sentinel-X Fund Share", "SNTL") Ownable(msg.sender) {
        fund = _fund;
    }

    function setFund(address _fund) external onlyOwner {
        fund = _fund;
    }

    function mint(address to, uint256 amount) external onlyFund {
        _mint(to, amount);
    }

    function burn(address from, uint256 amount) external onlyFund {
        _burn(from, amount);
    }
}
