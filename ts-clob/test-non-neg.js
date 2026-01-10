const { ClobClient } = require("@polymarket/clob-client");
const { Wallet } = require("ethers");

const PRIVATE_KEY = "f2603733a2748b8c4f56cd708bf8146621beb3b6c2e06be2101828166427f5a5";
const HOST = "https://clob.polymarket.com";
const CHAIN_ID = 137;

async function main() {
    const wallet = new Wallet(PRIVATE_KEY);
    console.log("Wallet:", wallet.address);

    const client = new ClobClient(HOST, CHAIN_ID, wallet);
    const creds = await client.deriveApiKey();
    client.creds = creds;

    // Get markets and find non-neg-risk one
    console.log("Finding non-neg-risk market...");
    const markets = await client.getMarkets();

    for (const m of markets.data.slice(0, 100)) {
        if (!m.neg_risk && m.active && m.tokens?.length > 0) {
            const token = m.tokens[0];
            console.log(`\nMarket: ${m.question.substring(0, 50)}...`);
            console.log(`neg_risk: ${m.neg_risk}`);
            console.log(`Token: ${token.token_id.substring(0, 40)}...`);

            try {
                const order = await client.createOrder({
                    tokenID: token.token_id,
                    price: 0.01,
                    size: 1,
                    side: "BUY",
                    feeRateBps: 0,
                });

                console.log("Order signed. Posting...");
                const result = await client.postOrder(order, "GTC");

                if (result.orderID) {
                    console.log("SUCCESS! Order:", result.orderID);
                    await client.cancelOrder(result.orderID);
                    console.log("Cancelled");
                } else if (result.error) {
                    console.log("Error:", result.error);
                } else {
                    console.log("Result:", result);
                }
                break;
            } catch (e) {
                const err = e.message || String(e);
                if (err.includes("orderbook") || err.includes("does not exist")) {
                    continue;  // Try next market
                }
                console.log("Error:", err.substring(0, 150));
                break;
            }
        }
    }
}

main().catch(console.error);
