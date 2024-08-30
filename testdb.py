from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
user = "neo4j"
password = "12345678"  # Replace <your_neo4j_password> with the password you set

driver = GraphDatabase.driver(uri, auth=(user, password))

def get_version(driver):
    with driver.session() as session:
        result = session.run("RETURN neo4j.version() AS version")
        for record in result:
            print(record["version"])

get_version(driver)
driver.close()
