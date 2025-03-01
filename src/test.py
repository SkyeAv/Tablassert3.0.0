from src.utils import (
    toolkit, arguments, logging, kgConfigParser,
    dataframe, export, nlp)
import unittest
import hashlib
import math
import os


class TestArguments(unittest.TestCase):

    def test_length(self) -> None:
        self.assertEqual(arguments.check_len_args([1]), None)
        with self.assertRaises(IndexError):
            arguments.check_len_args([1, 2, 3])

    def test_extensions(self) -> None:
        self.assertEqual(arguments.check_config_extension([".YmL"]), None)
        with self.assertRaises(ValueError):
            arguments.check_config_extension(["AamL"])
            arguments.check_config_extension(["YYML"])


class TestDataframe(unittest.TestCase):

    def test_four_zip(self) -> None:
        self.assertEqual(
            dataframe.four_zip("not_applicable"), 404)
        self.assertEqual(
            dataframe.four_zip("not applicable"), 404)
        self.assertNotEqual(
            dataframe.four_zip(0.12345), 404)

    def test_four_unzip(self) -> None:
        self.assertEqual(
            dataframe.four_unzip(404), "not_applicable")
        self.assertNotEqual(
            dataframe.four_unzip(0.12345), "not_applicable")

    def test_get_letters(self) -> None:
        """
        Tests the get_letters function in the dataframe module.

        Verifies that the function returns the correct string of uppercase
        letters for a given integer, similar to Excel column naming (e.g.,
        0 -> 'A', 25 -> 'Z', 26 -> 'AA').

        Args:
            None

        Returns:
            None
        """
        self.assertEqual(
            dataframe.get_letters(0), "A")
        self.assertEqual(
            dataframe.get_letters(25), "Z")
        self.assertEqual(
            dataframe.get_letters(26), "AA")
        self.assertEqual(
            dataframe.get_letters(701), "ZZ")
        self.assertEqual(
            dataframe.get_letters(-1), "")
        self.assertEqual(
            dataframe.get_letters(0), "A")

    def test_column_stylinator(self) -> None:
        with self.assertRaises(ValueError):
            dataframe.column_stylinator(None, "Random Parameter")

    def test_non_numeric_regexinator(self) -> None:
        self.assertEqual(
            dataframe.non_numeric_regexinator(
                "not_applicable"), "not_applicable")
        self.assertEqual(
            dataframe.non_numeric_regexinator(
                "9_ball"), "9")

    def test_first_elementinator(self) -> None:
        self.assertEqual(
            dataframe.first_elementinator([1, 2, 3]), 1)

    def test_remove_before_colon(self) -> None:
        self.assertEqual(
            dataframe.remove_before_colon("curie:thing"), "thing")
        self.assertEqual(
            dataframe.remove_before_colon("thing"), "thing")

    def test_biolink_it(self) -> None:
        self.assertEqual(
            dataframe.biolink_it("thing"), "biolink:thing")

    def test_join_domainsinator(self) -> None:
        self.assertEqual(
            dataframe.join_domainsinator(
                ["MESH:thing1", "MESH:thing2"]), "MESH:thing1,MESH:thing2")

    def test_null_zip(self) -> None:
        self.assertEqual(
            dataframe.null_zip("not_applicable"), 1e-10)
        self.assertEqual(
            dataframe.null_zip("not applicable"), 1e-10)
        self.assertEqual(
            dataframe.null_zip(0.12345), 0.12345)

    def test_null_unzip(self) -> None:
        self.assertEqual(
            dataframe.null_unzip(1e-10), "not_applicable")
        self.assertEqual(
            dataframe.null_unzip(0.12345), 0.12345)

    def test_log10(self) -> None:
        """
        Tests the log10 function in the dataframe module.

        Verifies that the function returns the correct value for
        positive and negative numbers, including zero.
        """
        self.assertAlmostEqual(
            dataframe.log10(10), 1.0,
            msg="log10 of 10 is not 1")
        self.assertAlmostEqual(
            dataframe.log10(0), math.log(0.0000000001, 10),
            msg="log10 of 0 is not -10")
        self.assertAlmostEqual(
            dataframe.log10(10.5), math.log(10.5, 10),
            msg="log10 of 10.5 is not log10(10.5)")
        self.assertAlmostEqual(
            dataframe.log10(1e-10), math.log(1e-10, 10),
            msg="log10 of 1e-10 is not log10(1e-10)")

    def test_p_zip(self) -> None:
        self.assertAlmostEqual(
            dataframe.p_zip(10), -1.0)

    def test_abslog10(self) -> None:
        self.assertAlmostEqual(
            dataframe.abslog10(-10), 1.0)

    def test_p_pentalty(self) -> None:
        self.assertEqual(
            dataframe.p_pentalty(1e-10), 0)
        self.assertEqual(
            dataframe.p_pentalty(1), -1)


class TestExport(unittest.TestCase):

    def test_edge_naminator(self) -> None:
        self.assertEqual(
            export.edge_naminator("KG3", "1.0.0"), "KG3_edges_v1.0.0.tsv")

    def test_node_naminator(self) -> None:
        self.assertEqual(
            export.node_naminator("KG3", "1.0.0"), "KG3_nodes_v1.0.0.tsv")


class TestKgConfigParser(unittest.TestCase):

    def test_check_kg_config(self) -> None:
        with self.assertRaises(ValueError):
            kgConfigParser.check_kg_config(["list"])

    def test_check_kg_subconfigs(self) -> None:
        """
        Tests the check_kg_subconfigs() function.

        Verifies that the check_kg_subconfigs() function raises a ValueError
        exception when the given kg_config is invalid, and does not raise an
        exception when the given kg_config is valid.
        """
        # Test that check_kg_subconfigs does not raise an exception when the
        # given kg_config is valid
        valid_kg_config = {
            "knowledge_graph_name": "test",
            "version_number": "1.0",
            "max_workers": 4,
            "p_value_cutoff": 0.05,
            "config_directories": [],
            "override_sqlite": "",
            "supplement_sqlite": "",
            "babel_sqlite": "",
            "kg2_sqlite": "",
            "progress_handler_timeout": 10,
            "predicates_sqlite": "",
            "confidence_model": "",
            "tfidf_vectorizer": ""
        }
        self.assertEqual(
            kgConfigParser.check_kg_subconfigs(valid_kg_config), None)
        # Test that check_kg_subconfigs raises a ValueError exception when the
        # given kg_config is invalid
        invalid_kg_config = {
            "knowledge_graph_name": "test",
            "version_number": "1.0",
            "max_workers": 4,
            "config_directories": [],
            "override_sqlite": "",
            "supplement_sqlite": "",
            "babel_sqlite": "",
            "kg2_sqlite": "",
            "progress_handler_timeout": 10,
            "predicates_sqlite": "",
            "confidence_model": "",
        }
        with self.assertRaises(ValueError):
            kgConfigParser.check_kg_subconfigs(invalid_kg_config)

    def test_check_int(self) -> None:
        self.assertEqual(
            kgConfigParser.check_int(1, ""), None)
        with self.assertRaises(ValueError):
            kgConfigParser.check_int(["list"], "")

    def test_check_float(self) -> None:
        self.assertEqual(
            kgConfigParser.check_float(1.0, ""), None)
        with self.assertRaises(ValueError):
            kgConfigParser.check_float(["list"], "")

    def test_check_str(self) -> None:
        self.assertEqual(
            kgConfigParser.check_str("str", ""), None)
        with self.assertRaises(ValueError):
            kgConfigParser.check_str(["list"], "")

    def test_check_pkl(self) -> None:
        with self.assertRaises(ValueError):
            kgConfigParser.check_pkl(r"file.xlsx", "")


class TestLogging(unittest.TestCase):

    def test_newline_cleaner(self) -> None:
        self.assertEqual(
            logging.newline_cleaner("Hello\nWorld"), "HelloWorld")
        self.assertEqual(
            logging.newline_cleaner("HelloWorld"), "HelloWorld")

    def test_get_log(self) -> None:
        logger = logging.get_log()
        self.assertEqual(
            os.path.basename(
                logger.handlers[0].baseFilename), "tablassert.log")


class TestNlp(unittest.TestCase):

    def test_tokenize_it(self) -> None:
        self.assertEqual(
            nlp.tokenize_it(
                "This is a test sentence"), "This a is sentence test")

    def test_nonword_regex(self) -> None:
        self.assertEqual(
            nlp.nonword_regex("Hello, World! 123"), "HelloWorld123")

    def test_hash_it(self) -> None:
        self.assertEqual(
            nlp.hash_it("hello!@#world"), hashlib.sha1(
                ("helloworld").encode("utf-8")).hexdigest())


class TestToolkit(unittest.TestCase):

    def test_fast_extension(self) -> None:
        self.assertEqual(
            toolkit.fast_extension("file.txt"), ".txt")
        self.assertNotEqual(
            toolkit.fast_extension("file.yaml"), ".yaml")

    def test_get_dirname(self) -> None:
        self.assertEqual(
            toolkit.get_dirname("/path/to/file.txt"), "/path/to")

    def test_get_filename_no_ext(self) -> None:
        self.assertEqual(
            toolkit.get_filename_no_ext(
                "example.txt"), "example")
        self.assertEqual(
            toolkit.get_filename_no_ext(
                "example"), "example")

    def test_tsv_it(self) -> None:
        self.assertEqual(
            toolkit.tsv_it(
                "example"), "example.tsv")

    def test_slash_cleaner(self) -> None:
        self.assertEqual(
            toolkit.slash_cleaner("hello/world"), "helloworld")
        self.assertEqual(
            toolkit.slash_cleaner("helloworld"), "helloworld")


def main() -> None:
    """
    Runs all the unit tests in the given test classes.

    This function does not take any arguments and returns nothing.
    It creates a suite of tests from a list of test classes and runs them.
    Note: unittest.main() does not work with the setup.py
    """
    test_classes: list[type] = [
        TestArguments, TestDataframe, TestExport,
        TestKgConfigParser, TestLogging, TestNlp,
        TestToolkit]
    suite: unittest.TestSuite = unittest.TestSuite()
    for test_class in test_classes:
        suite.addTests(
            unittest.defaultTestLoader.loadTestsFromTestCase(test_class))
    unittest.TextTestRunner().run(suite)


if __name__ == "__main__":
    main()
