import subprocess
import threading


class Command:
    """
    Enables to run subprocess commands in a different thread
    with TIMEOUT option!
    Based on jcollado's solution:
    http://stackoverflow.com/questions/1191374/subprocess-with-timeout/4825933#4825933
    """

    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None

    def run(self, timeout=0, **kwargs):
        def target(**kwargs):
            self.process = (
                subprocess.Popen(  # pylint: disable=consider-using-with
                    self.cmd, **kwargs
                )
            )
            self.process.communicate()

        thread = threading.Thread(target=target, kwargs=kwargs)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            self.process.terminate()
            thread.join()

        return self.process.returncode
