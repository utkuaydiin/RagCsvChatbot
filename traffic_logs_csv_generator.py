import csv
import random
import faker

# Create a Faker instance
fake = faker.Faker()

# Function to generate a random timestamp
def generate_timestamp():
    return fake.date_time_between(start_date="-1y", end_date="now").strftime("%Y-%m-%d %H:%M:%S%z")

# Function to generate a random IP address
def generate_ip():
    return fake.ipv4()

# Function to generate a random HTTP method
def generate_method():
    return random.choice(["GET", "POST", "PUT", "DELETE"])

# Function to generate a random URL path
def generate_url():
    paths = ["/index.html", "/about.html", "/contact.html", "/products.html", "/services.html", "/login.html"]
    return random.choice(paths)

# Function to generate a random status code
def generate_status():
    return random.choice([200, 201, 301, 400, 401, 403, 404, 500, 503])

# Function to generate a random size
def generate_size():
    return random.randint(100, 5000)

# Function to generate a random User-Agent
def generate_user_agent():
    return fake.user_agent()

# Generate the data
data = [
    [generate_ip(), generate_timestamp(), generate_method(), generate_url(), generate_status(), generate_size(), generate_user_agent()]
    for _ in range(10)  # Change 10 to however many rows you want
]

# Write the data to a CSV file
with open("log_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["IP Address", "Timestamp", "Method", "URL", "Status", "Size", "User Agent"])
    # Write the data
    writer.writerows(data)

print("CSV file 'log_data.csv' generated successfully.")
