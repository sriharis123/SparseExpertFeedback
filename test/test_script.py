#stack overflow
import sched, time
s = sched.scheduler(time.time, time.sleep)
def do_something(sc): 
    print("Doing stuff...", time.time())
    # do your stuff
    sc.enter(1, 1, do_something, (sc,))

s.enter(1, 1, do_something, (s,))
s.run()