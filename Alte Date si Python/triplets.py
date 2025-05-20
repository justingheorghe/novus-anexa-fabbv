import json

# Generate all positive integer quadruplets (a,b,c,d) such that a+b+c+d=100
# and a, b, c, d are all divisible by 5.
quadruplets = [
    (a, b, c, 100 - a - b - c)
    for a in range(5, 90, 5)  # a in [5, ..., 85], ensuring a,b,c,d can be >= 5
    for b in range(5, 95 - a, 5)  # b in [5, ..., 90-a]
    for c in range(5, 100 - a - b, 5) # c in [5, ..., 95-a-b], ensures d >= 5
    # d = 100 - a - b - c will also be a multiple of 5 and >= 5
]

# Write to JSON file
with open("quadruplets_divisible_by_5.json", "w") as f:
    json.dump({"quadruplets_divisible_by_5": quadruplets}, f, indent=4)

print(f"Saved {len(quadruplets)} quadruplets (all elements divisible by 5) to quadruplets_divisible_by_5.json")
