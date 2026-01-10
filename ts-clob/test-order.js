const { ClobClient } = require("@polymarket/clob-client");
const { Wallet } = require("ethers");

const PRIVATE_KEY = "f2603733a2748b8c4f56cd708bf8146621beb3b6c2e06be2101828166427f5a5";
const HOST = "https://clob.polymarket.com";
const CHAIN_ID = 137;

async function main() {
    // Create wallet
    const wallet = new Wallet(PRIVATE_KEY);
    console.log("Wallet address:", wallet.address);

    // Create client (EOA - no funder needed)
    const client = new ClobClient(HOST, CHAIN_ID, wallet);

    // Get/derive API credentials
    console.log("Deriving API key...");
    const creds = await client.deriveApiKey();
    client.creds = creds;
    console.log("API key:", creds.key.substring(0, 25) + "...");

    // Check balance
    console.log("\nChecking balance...");
    const balanceAllowance = await client.getBalanceAllowance({ asset_type: "COLLATERAL" });
    console.log("Balance:", (parseInt(balanceAllowance.balance) / 1e6).toFixed(2), "USDC");

    // Get current BTC 15-min market token from our server
    console.log("\nFetching BTC 15-min market...");
    const resp = await fetch("http://localhost:8000/api/markets-15m");
    const markets = await resp.json();
    const btc = markets.active?.BTC;

    if (!btc) {
        console.log("No BTC market found");
        return;
    }

    console.log("Market:", btc.question.substring(0, 50) + "...");
    console.log("Token:", btc.up_token_id.substring(0, 40) + "...");

    // Try to place order
    // NOTE: Don't force negRisk - let the library auto-detect via the /neg-risk endpoint
    // Forcing negRisk=true when the API returns false causes "invalid signature" errors
    console.log("\nPlacing test order...");
    try {
        const order = await client.createOrder({
            tokenID: btc.up_token_id,
            price: 0.02,
            size: 5,  // Minimum size is 5 shares
            side: "BUY",
        });  // Let library auto-detect negRisk and feeRateBps

        console.log("Order signed. Posting...");
        const result = await client.postOrder(order, "GTC");
        console.log("SUCCESS!", result);

        if (result.orderID) {
            await client.cancelOrder(result.orderID);
            console.log("Cancelled");
        }
    } catch (e) {
        if (e.message?.includes("invalid signature")) {
            console.log("INVALID SIGNATURE - TypeScript client also fails");
        } else if (e.message?.includes("insufficient")) {
            console.log("Signature OK! But insufficient funds (expected)");
        } else {
            console.log("Error:", e.message || e);
        }
    }
}

main().catch(console.error);
