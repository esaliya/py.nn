if __name__ == '__main__':
    A = 84225
    B = 128

    # Now I want to find a number P s.t. P is an integer and
    # P = A-x/B+y where x is minimized. Also, y could be plus or minus

    # actually let's restrict y to be positive too, so we want to B to be >= 128

    xrange = range(0, A + 1)
    yrange = range(0, (A - B) + 1)

    xmin = A
    yfinal = 0

    for x in xrange:
        for y in yrange:
            Pmod = (A - x) % (B + y)
            if Pmod == 0:
                if x < xmin:
                    xmin = x
                    yfinal = y
                    print(xmin, yfinal, (A - x) / (B + y))

