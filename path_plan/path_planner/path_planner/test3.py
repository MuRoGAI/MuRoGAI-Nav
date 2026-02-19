def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split("-"))
    return h * 3600 + m * 60 + s


def seconds_to_time(total_seconds):
    sign = "-" if total_seconds < 0 else ""
    total_seconds = abs(total_seconds)

    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60

    return f"{sign}{h:02d}-{m:02d}-{s:02d}"


# ==============================
# Inputs (EDIT HERE)
# ==============================

time1 = "00-10-47"
time2 = "00-06-00"
operation = "-"   # "+" or "-"

# ==============================
# Calculation
# ==============================

sec1 = time_to_seconds(time1)
sec2 = time_to_seconds(time2)

if operation == "+":
    result = sec1 + sec2
elif operation == "-":
    result = sec1 - sec2
else:
    raise ValueError("Invalid operation. Use '+' or '-'.")

print("Result:", seconds_to_time(result))
