"""
razorpay_client.py — Real Razorpay API integration for VAADA
Generates actual test-mode payment links via Razorpay API
"""

import os
import requests
import json
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta

# ── Test mode credentials ─────────────────────────────────────
KEY_ID     = os.environ.get("RAZORPAY_KEY_ID")
KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET")
BASE_URL   = "https://api.razorpay.com/v1"
AUTH       = HTTPBasicAuth(KEY_ID, KEY_SECRET)
HEADERS    = {"Content-Type": "application/json"}


def create_payment_link(
    amount: float,
    customer_name: str = "Customer",
    description: str = "EMI Recovery - VAADA Collections",
    accept_partial: bool = False,
    min_partial: float = None,
    intent: str = "promise_to_pay",
    dpd: int = 0,
    expire_hours: int = 24
) -> dict:
    """
    Create a real Razorpay payment link via API.
    Amount in INR (converted to paise internally).
    """
    amount_paise = int(amount * 100)

    expire_by = int(
        (datetime.now() + timedelta(hours=expire_hours)).timestamp()
    )

    payload = {
        "amount":          amount_paise,
        "currency":        "INR",
        "accept_partial":  accept_partial,
        "description":     description,
        "customer": {
            "name":    customer_name,
           "contact": "+918369284390",
            "email":   "customer@vaada.ai"
        },
        "notify": {
            "sms":   False,
            "email": False
        },
        "reminder_enable": False,
        "expire_by":       expire_by,
        "notes": {
            "source":  "VAADA",
            "intent":  intent,
            "dpd":     str(dpd),
            "version": "1.0"
        },
        "callback_url":    "https://github.com/singhhnitin/vaada",
        "callback_method": "get"
    }

    if accept_partial and min_partial:
        payload["first_min_partial_amount"] = int(min_partial * 100)

    try:
        response = requests.post(
            f"{BASE_URL}/payment_links",
            auth=AUTH,
            headers=HEADERS,
            data=json.dumps(payload),
            timeout=10
        )
        data = response.json()

        if response.status_code == 200:
            return {
                "success":    True,
                "link_id":    data.get("id"),
                "short_url":  data.get("short_url"),
                "amount":     amount,
                "currency":   "INR",
                "status":     data.get("status"),
                "expires_at": datetime.fromtimestamp(expire_by).strftime("%Y-%m-%d %H:%M"),
                "raw":        data
            }
        else:
            return {
                "success": False,
                "error":   data.get("error", {}).get("description", "API error"),
                "code":    response.status_code
            }

    except Exception as e:
        return {
            "success": False,
            "error":   str(e)
        }


def create_partial_payment_link(
    total_amount: float,
    partial_amount: float,
    customer_name: str = "Customer",
    intent: str = "partial_payment",
    dpd: int = 0
) -> dict:
    """Create a partial payment link."""
    return create_payment_link(
        amount         = total_amount,
        customer_name  = customer_name,
        description    = f"Partial EMI Recovery - VAADA (Rs{partial_amount:.0f} now)",
        accept_partial = True,
        min_partial    = partial_amount,
        intent         = intent,
        dpd            = dpd
    )


def get_payment_link_status(link_id: str) -> dict:
    """Check status of an existing payment link."""
    try:
        response = requests.get(
            f"{BASE_URL}/payment_links/{link_id}",
            auth=AUTH,
            headers=HEADERS,
            timeout=10
        )
        data = response.json()
        return {
            "link_id":    link_id,
            "status":     data.get("status"),
            "amount":     data.get("amount", 0) / 100,
            "amount_paid":data.get("amount_paid", 0) / 100,
            "short_url":  data.get("short_url"),
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print("=== VAADA Razorpay Integration Test ===\n")

    # Test 1: Full payment link
    print("Test 1: Full payment link for Rs5000...")
    result = create_payment_link(
        amount        = 5000,
        customer_name = "Rahul Singh",
        description   = "EMI Recovery - VAADA Collections",
        intent        = "promise_to_pay",
        dpd           = 8
    )

    if result["success"]:
        print(f"SUCCESS")
        print(f"  Link ID  : {result['link_id']}")
        print(f"  URL      : {result['short_url']}")
        print(f"  Amount   : Rs{result['amount']}")
        print(f"  Status   : {result['status']}")
        print(f"  Expires  : {result['expires_at']}")
    else:
        print(f"FAILED: {result['error']}")

    print()

    # Test 2: Partial payment link
    print("Test 2: Partial payment link Rs6000 of Rs12000...")
    result2 = create_partial_payment_link(
        total_amount   = 12000,
        partial_amount = 6000,
        customer_name  = "Amit Kumar",
        intent         = "partial_payment",
        dpd            = 20
    )

    if result2["success"]:
        print(f"SUCCESS")
        print(f"  Link ID  : {result2['link_id']}")
        print(f"  URL      : {result2['short_url']}")
        print(f"  Status   : {result2['status']}")
    else:
        print(f"FAILED: {result2['error']}")
