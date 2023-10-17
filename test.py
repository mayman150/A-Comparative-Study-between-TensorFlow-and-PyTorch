
def test(variable):
    if variable.a > 10:
        variable.init()
        if variable.c() and variable.f():
            print("OK")
        else:
            print("Not OK")
    elif variable.b < 2:
        return True
    else:
        if variable.d:
            return False