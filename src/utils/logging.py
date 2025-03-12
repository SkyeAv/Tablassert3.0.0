import logging
import re


def newline_cleaner(thing_to_clean: str) -> str:
    return re.sub(r"\n", "", str(thing_to_clean))


def get_log(log_path: str = "tablassert.log") -> logging.Logger:
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s\t%(levelname)s\t%(message)s",
            filename=log_path,
            filemode="w"
        )
    return logging.getLogger()


def log_mapped_edge(what: str, became_what: list[str], method: str) -> None:
    logger = get_log()
    mapping_message = f"""{newline_cleaner(what)} became {
            newline_cleaner(became_what)}\t{newline_cleaner(method)}"""
    logger.info(mapping_message)


def log_dropped_edge(what_got_dropped: str, reason: str) -> None:
    logger = get_log()
    dropping_message = f"""dropped {
        newline_cleaner(what_got_dropped)}\t{newline_cleaner(reason)}"""
    logger.warning(dropping_message)


def log_current_table(table_name: str) -> None:
    logger = get_log()
    logger.info(f"Beginning {newline_cleaner(table_name)}")


def log_thing(thing: str) -> None:
    logger = get_log()
    logger.info(f"{newline_cleaner(thing)}")


def log_error(msg: object) -> None:
    logger = get_log()
    logger.error(newline_cleaner(msg))


def log_slow_query(val: str, db: str, err: str) -> None:
    logger = get_log()
    logger.warning(newline_cleaner(f"Slow Query: {val}\t{db}\t{err}"))
