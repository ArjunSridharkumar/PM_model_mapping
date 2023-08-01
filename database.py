import sqlite3

# Function to create a SQLite database connection
def create_connection(database_name):
    try:
        connection = sqlite3.connect(database_name)
        return connection
    except sqlite3.Error as e:
        print(e)
        return None

# Create the resumes table in the database
def create_resumes_table(connection):
    if connection is not None:
        create_table_query = """
            CREATE TABLE IF NOT EXISTS resumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                gpa REAL,
                email TEXT,
                experience TEXT,
                previous_companies TEXT,
                schools TEXT,
                resume_text TEXT
            );
        """
        try:
            cursor = connection.cursor()
            cursor.execute(create_table_query)
            connection.commit()
            cursor.close()
        except sqlite3.Error as e:
            print(e)

# Function to store resume and information in the database
def insert_resume(connection, candidate_info):
    insert_query = """
        INSERT INTO resumes (name, gpa, email, experience, previous_companies, schools, resume_text)
        VALUES (?, ?, ?, ?, ?, ?, ?);
    """
    cursor = connection.cursor()
    cursor.execute(insert_query, (
        candidate_info["name"],
        candidate_info["gpa"],
        candidate_info["email"],
        candidate_info["experience"],
        ", ".join(candidate_info["previous_companies"]),
        ", ".join(candidate_info["schools"]),
        candidate_info["resume_text"]
    ))
    connection.commit()
    cursor.close()

# Function to retrieve all resumes from the database
def get_all_resumes(connection):
    select_query = "SELECT * FROM resumes;"
    cursor = connection.cursor()
    cursor.execute(select_query)
    resumes = cursor.fetchall()
    cursor.close()
    return resumes