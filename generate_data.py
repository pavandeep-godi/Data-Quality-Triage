"""Creates data/good_data.csv and data/bad_data.csv (500 rows x 20 columns each)."""
import pathlib
import numpy as np
import pandas as pd

N = 500
COUNTRIES = {"USA": ["New York", "Chicago", "Austin"], "UK": ["London", "Leeds", "Bristol"],
             "India": ["Mumbai", "Delhi", "Pune"], "Germany": ["Berlin", "Munich", "Hamburg"],
             "Canada": ["Toronto", "Vancouver", "Montreal"]}
PRODUCTS = {"Electronics": ["Headphones", "Keyboard", "Webcam"], "Home": ["Lamp", "Blender", "Kettle"],
            "Fitness": ["Yoga Mat", "Dumbbells", "Skipping Rope"], "Books": ["Novel", "Cookbook", "Atlas"]}
FIRST = ["Ava", "Liam", "Noah", "Mia", "Ethan", "Zoe", "Arjun", "Priya", "Lena", "Omar", "Sofia", "Kai"]
LAST = ["Smith", "Patel", "Müller", "Brown", "Singh", "Khan", "Garcia", "Lee", "Wong", "Taylor"]


def make_good(seed):
    r = np.random.default_rng(seed)
    country = r.choice(list(COUNTRIES), N)
    cat = r.choice(list(PRODUCTS), N)
    qty = r.integers(1, 11, N)
    price = np.round(r.uniform(5, 500, N), 2)
    disc = r.choice([0, 5, 10, 15, 20, 25, 30], N)
    order = pd.to_datetime("2025-01-01") + pd.to_timedelta(r.integers(0, 365, N), unit="D")
    ship = order + pd.to_timedelta(r.integers(1, 6, N), unit="D")
    total = np.round(qty * price * (1 - disc / 100), 2)
    first, last = r.choice(FIRST, N), r.choice(LAST, N)
    cid = r.integers(1000, 1200, N)
    return pd.DataFrame({
        "order_id": np.arange(100001, 100001 + N),
        "customer_id": cid,
        "customer_name": [f"{a} {b}" for a, b in zip(first, last)],
        "email": [f"{a.lower()}.{b.lower().replace('ü', 'u')}{c}@example.com" for a, b, c in zip(first, last, cid)],
        "country": country,
        "city": [r.choice(COUNTRIES[c]) for c in country],
        "product_category": cat,
        "product_name": [r.choice(PRODUCTS[c]) for c in cat],
        "quantity": qty,
        "unit_price": price,
        "discount_pct": disc,
        "total_amount": total,
        "order_date": order.strftime("%Y-%m-%d"),
        "ship_date": ship.strftime("%Y-%m-%d"),
        "status": r.choice(["Delivered", "Shipped", "Processing", "Returned"], N, p=[.6, .2, .15, .05]),
        "payment_method": r.choice(["Card", "UPI", "PayPal", "Bank Transfer"], N),
        "rating": r.integers(1, 6, N),
        "is_member": r.choice(["True", "False"], N),
        "shipping_cost": np.where(total > 100, 0.0, np.round(r.uniform(4, 12, N), 2)),
        "delivery_days": r.integers(1, 8, N),
    })


def make_bad(seed):
    r = np.random.default_rng(seed + 1)
    df = make_good(seed).astype(object)
    pick = lambda k: r.choice(N, k, replace=False)
    # nulls in 3 columns (city also gets whitespace-only "empty" strings)
    df.loc[pick(45), "email"] = np.nan
    df.loc[pick(25), "city"] = np.nan
    df.loc[pick(15), "city"] = "  "
    df.loc[pick(50), "rating"] = np.nan
    # anomalies in 1 column: absurd unit prices (total_amount stays correct, so it disagrees)
    idx = pick(12)
    df.loc[idx, "unit_price"] = np.round(df.loc[idx, "unit_price"].astype(float) * r.uniform(30, 90, 12), 2)
    # wrong data type in 1 column: text inside a numeric column
    df.loc[pick(15), "quantity"] = r.choice(["three", "N/A", "ten", "two", "abc", "5 units"], 15)
    return df


if __name__ == "__main__":
    out = pathlib.Path("data")
    out.mkdir(exist_ok=True)
    make_good(42).to_csv(out / "good_data.csv", index=False)
    make_bad(7).to_csv(out / "bad_data.csv", index=False)
    print("wrote", list(out.iterdir()))
