import sqlite3
import datetime 
import configparser


class NextdoorDBConnection:
    """
    Class handling useful read/write behavior to our project SQLite Database.
    """

    def __init__(self, path: str, table: str):
        """
        param path: Path to .db file
        param table: Name of SQL table to read/write
        """
        self.conn = sqlite3.connect(path + 'nextdoor.db')
        self.cursor = self.conn.cursor() 
        self.table = table


    def get_prior_ids(self) -> list:
        """
        Return list of all post ids that have previously been recommended by the engine.
        """
        query = "SELECT id from {}".format(self.table)
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        return [row[0] for row in result]


    def get_consecutive_days(self, id: int) -> int:
        """
        For a previously-recommended post, return the number of days in a row it has been recommended.
        """

        query = "SELECT consecutiveDays FROM {} WHERE id = ?".format(self.table)
        params = (id,)

        self.cursor.execute(query)
        result = self.cursor.fetchall()
        return int(result[0][0])
    

    def add_row_to_db(self, id: int, author: str) -> None:
        """
        Write a new row to the database table for a newly-recommended post.
        """

        date = datetime.date.today()
        query = "INSERT INTO {} (id, firstDate, recentDate, consecutiveDays, author) VALUES (?, ?, ?, ?, ?)".format(self.table)
        params = (id, date, date, 1, author)

        self.cursor.execute(query, params)
        self.conn.commit()


    def update_row_in_db(self, id: int, date: datetime.date) -> None:
        """
        Update an existing row in the database table for a post that has been recommended again.
        """

        num_days = self.get_consecutive_days(id)
        query = "UPDATE  {} SET recentDate = ?, consecutiveDays = ? WHERE id = ?".format(self.table)
        params = (date, num_days + 1, id)

        self.cursor.execute(query, params)
        self.conn.commit()


    def reset_streak(self, id: int) -> None:
        """
        Reset the number of consecutive days field to zero for a post that is no longer being recommended. 
        """

        query = "UPDATE {} SET consecutiveDays = 0 WHERE id = ?".format(self.table)
        params = (id,)
        self.cursor.execute(query, params)
        self.conn.commit()


    def __del__(self):
        self.cursor.close()
        self.conn.close()


def append_to_log(date: datetime.datetime, message: str, destination: str):
    """
    Write a new entry to the specified log file.

    param date: Today's datetime object
    param message: Status message to be written to log
    param destination: Path to destination log file
    """

    date_str = date.strftime("%m-%d-%y")
    input_str = date_str + "|" + message + "\n"
    with open(destination, 'a') as fp:
        fp.write(input_str)


def get_link_from_id(id: int) -> str:
    """
    Return link to Nextdoor post with given id.
    """
    return "https://nextdoor.com/news_feed/?post=" + str(id)


def get_smtp_settings(path: str) -> tuple:
    """
    Read configuration file and return SMTP settings.
    """

    with open(path, 'r') as fp:
        config = configparser.ConfigParser()
        
        # Note: Unlike Java, Python requires config files to have section headings
        # To make our config files readable in both languages, we add a "dummy section"
        config_str = '[dummy_section]\n' + fp.read()
        config.read_string(config_str)

        server = config['dummy_section']['Server']
        port = config['dummy_section']['Port']
        sender_email = config['dummy_section']['SenderEmail']
        sender_password = config['dummy_section']['SenderPassword']
        receiver_email = config['dummy_section']['ReceiverEmail']

        return (server, int(port), sender_email, sender_password, receiver_email)





