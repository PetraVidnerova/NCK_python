from torch.multiprocessing import Process, Queue

def worker(name, evalfunc, querries, answers, device):
    while True:
        querry = querries.get()
        answer = evalfunc(querry, device)
        answers.put(answer)


    
class Pool:
    """ Pool of workers. Workers consume tasks from the querries queue and 
        feed answers to the answers queue.
    """

    
    def __init__(self, processors, evalfunc, devices):
        self.querries = Queue(1000)
        self.answers = Queue(1000)

        self.workers = []
        for i in range(processors):
             worker_i = Process(target=worker, args=(i, evalfunc, self.querries, self.answers, devices[i]))
             self.workers.append(worker_i)
             worker_i.start()        # Launch worker() as a separate python process

    def putQuerry(self, querry):
        self.querries.put(querry)

    def getAnswer(self):
        return self.answers.get()
             

    def close(self):
        for w in self.workers:
            w.terminate()
        #print("pool killed")
        
