a = int(input("今年は西暦何年"))

if a % 4 == 0:
    if a % 100 != 0 or a % 400 == 0:
        print("True")
    else:
        print("False")
else:
    print("False")
