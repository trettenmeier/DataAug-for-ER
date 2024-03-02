import os
from typing import TYPE_CHECKING

import joblib
import luigi
from luigi.task import flatten

if TYPE_CHECKING:
    from typing import Any


class LuigiBaseTask(luigi.Task):
    def run(self) -> None:
        """
        Will need to load correct dataframe that meets all requirements for current task. Then transforms with
        associated transformer. Writes the result dataframe.
        """
        raise NotImplementedError()

    def output(self) -> luigi.local_target:
        raise NotImplementedError()

    def complete(self) -> bool:
        """
        This method changes the default behavior of luigi to assume that if a task is called and is complete,
        all required tasks are complete as well. The desired behavior is that all required tasks are checked and run
        again as well as the tasks depending on them.

        Returns
        -------
        bool
            True if task and requirements are complete, False otherwise
        """
        outputs = flatten(self.output())
        if not all(map(lambda output: output.exists(), outputs)):
            return False
        for task in flatten(self.requires()):
            if not task.complete():
                for output in outputs:
                    if output.exists():
                        output.remove()
                return False
        return True

    def invalidate(self) -> str:
        """
        Deletes the output-file if it exists so when the task gets called it will be run again.

        Returns
        -------
        str
            either message that output was deleted or message that there was nothing to delete
        """
        try:
            os.remove(self.output().path)
            return f"Deleted {self.output().path}"
        except OSError:
            return f"Nothing to delete, {self.output().path} does not exist"

    def make_output_target(self, output_path: str, filename: str) -> luigi.LocalTarget:
        """
        Small helper that creates a luigi output target out of a filename and a path.

        Parameters
        ----------
        output_path: str
            output path name
        filename: str
            output file's name

        Returns
        -------
        LocalTarget
            luigi file system target
        """
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, filename)
        return luigi.LocalTarget(output_path, format=luigi.format.Nop)

    def write(self, file: 'Any') -> None:
        """
        Writes the given variable on the disk (to the path + filename specified in output()).

        Parameters
        ----------
        file
            file to be saved as a joblib
        """
        with open(self.output().path, "w") as f:
            f.write(file)
