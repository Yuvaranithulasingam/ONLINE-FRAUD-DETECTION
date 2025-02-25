// script.js
document.getElementById("paymentForm").addEventListener("submit", function(event) {
    event.preventDefault();

    let cardNumber = document.getElementById("cardNumber").value;
    let expiry = document.getElementById("expiry").value;
    let cvv = document.getElementById("cvv").value;
    let amount = document.getElementById("amount").value;

    if (!validateInput(cardNumber, expiry, cvv, amount)) {
        alert("Please enter valid details!");
        return;
    }

    // Simulate fraud detection API call
    checkFraud({ cardNumber, expiry, cvv, amount });
});

function validateInput(cardNumber, expiry, cvv, amount) {
    return cardNumber.length === 16 && expiry.match(/^\d{2}\/\d{2}$/) && cvv.length === 3 && amount > 0;
}

function checkFraud(transactionData) {
    // Simulate API call
    fetch("https://your-backend-api.com/check_fraud", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(transactionData)
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = data.isFraud ? "🚨 Fraud Detected!" : "✅ Transaction is Safe!";
    })
    .catch(error => console.error("Error:", error));
}
