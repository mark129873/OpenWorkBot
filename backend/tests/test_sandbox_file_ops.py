import importlib.util
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path


if sys.platform.startswith("linux"):
    source = Path(os.environ.get("SANDBOX_FILE_OPS_PATH", Path(__file__).resolve().parents[1]
                                 / "app/community/aio_sandbox/file_ops.py"))
    spec = importlib.util.spec_from_file_location("sandbox_file_ops", source)
    file_ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(file_ops)


@unittest.skipUnless(sys.platform.startswith("linux"), "需要 Linux openat2")
class SandboxFileOpsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "workspace"
        self.root.mkdir()
        file_ops.ROOT = str(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def call(self, operation, **args):
        return file_ops.execute({"operation": operation, "args": args, "windows_newline": False})

    def test_read_write_edit_glob_and_trash(self):
        self.call("write_file", path="目录/a.txt", content="第一行\nold\n")
        self.assertIn("1: 第一行", self.call("read_file", path="目录/a.txt"))
        version = self.call("version", path="目录/a.txt")
        self.assertEqual(len(version), 64)
        self.call("edit_file", path="目录/a.txt", old_text="old", new_text="new")
        self.assertNotEqual(version, self.call("version", path="目录/a.txt"))
        self.assertIn("目录/a.txt", self.call("glob", pattern="**/*.txt"))
        self.call("delete_file", path="目录/a.txt")
        self.assertFalse((self.root / "目录/a.txt").exists())
        self.assertEqual(len(list((self.root / ".rubbish").iterdir())), 1)

    def test_rejects_outside_links_and_accepts_internal_links(self):
        (self.root / "inside").write_text("inside", encoding="utf-8")
        outside = self.base / "outside"
        outside.write_text("secret", encoding="utf-8")
        (self.root / "internal").symlink_to("inside")
        (self.root / "external").symlink_to(outside)
        self.assertIn("inside", self.call("read_file", path="internal"))
        with self.assertRaises(OSError):
            self.call("read_file", path="external")
        with self.assertRaises(OSError):
            self.call("write_file", path="external", content="changed")
        self.assertEqual(outside.read_text(encoding="utf-8"), "secret")

    def test_trash_directory_link_cannot_move_file_outside(self):
        outside = self.base / "outside"
        outside.mkdir()
        (self.root / "data").write_text("safe", encoding="utf-8")
        (self.root / ".rubbish").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(OSError):
            self.call("delete_file", path="data")
        self.assertTrue((self.root / "data").exists())
        self.assertEqual(list(outside.iterdir()), [])

    def test_concurrent_link_replacement_never_reads_outside(self):
        (self.root / "inside").write_text("safe", encoding="utf-8")
        outside = self.base / "outside"
        outside.write_text("secret", encoding="utf-8")
        link = self.root / "link"
        link.symlink_to("inside")
        stop = threading.Event()

        def swap():
            temporary = self.root / "link-next"
            while not stop.is_set():
                for target in ("inside", str(outside)):
                    temporary.symlink_to(target)
                    temporary.replace(link)

        thread = threading.Thread(target=swap)
        thread.start()
        try:
            for _ in range(500):
                try:
                    result = self.call("read_file", path="link")
                except OSError:
                    continue
                self.assertNotIn("secret", result)
        finally:
            stop.set()
            thread.join()


if __name__ == "__main__":
    unittest.main()
