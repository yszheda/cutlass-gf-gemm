prim_poly = 0x1D
e = 1
gf_log = [0] * 256
gf_exp = [0] * 768

for log in range(255):
    print(f"log={log}, e={e}")
    if e < 256:
        gf_log[e] = log
        gf_exp[log] = e
        gf_exp[log + 255] = e
        gf_exp[log + 510] = e
    next_e = (e << 1) ^ (prim_poly if (e & 0x80) else 0)
    e = next_e & 0xFF
    if e == 0:
        print("e became 0!")
        break
