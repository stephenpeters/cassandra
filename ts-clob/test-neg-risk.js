require('dotenv').config({ path: '../backend/.env' });
const { ClobClient } = require("@polymarket/clob-client");
const { Wallet } = require("ethers");

const PRIVATE_KEY = process.env.POLYMARKET_PRIVATE_KEY;
const PROXY_WALLET = process.env.POLYMARKET_PROXY_WALLET || "0x21f6163c35B3B2523a4db1cf61B33E55b8e071b1";
const HOST = "https://clob.polymarket.com";
const CHAIN_ID = 137;

if (!PRIVATE_KEY) {
    console.error("Error: POLYMARKET_PRIVATE_KEY not found in ../backend/.env");
    process.exit(1);
}

async function main() {
    const wallet = new Wallet(PRIVATE_KEY);
    console.log("EOA Wallet:", wallet.address);
    console.log("Proxy Wallet:", PROXY_WALLET);

    // Create client with proxy wallet (signature type 2)
    const client = new ClobClient(HOST, CHAIN_ID, wallet, undefined, 2, PROXY_WALLET);
    const creds = await client.deriveApiKey();
    client.creds = creds;
    console.log("API Key:", creds.key.substring(0, 30) + "...");

    // Get BTC 15-min market from local server
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
    console.log("\nPlacing test order (auto-detect negRisk)...");
    try {
        const order = await client.createOrder({
            tokenID: btc.up_token_id,
            price: 0.02,
            size: 5,  // Minimum size is 5 shares
            side: "BUY",
        });  // Let library auto-detect negRisk and feeRateBps

        console.log("Order created. Posting...");
        console.log("Order maker:", order.order?.maker);
        console.log("Order signer:", order.order?.signer);
        
        const result = await client.postOrder(order, "GTC");
        console.log("SUCCESS:", result);

        if (result.orderID) {
            await client.cancelOrder(result.orderID);
            console.log("Cancelled");
        }
    } catch (e) {
        const err = e.message || String(e);
        if (err.includes("insufficient")) {
            console.log("SIGNATURE OK! Funds error:", err.substring(0, 80));
        } else {
            console.log("Error:", err.substring(0, 150));
        }
    }
}

main().catch(console.error);
