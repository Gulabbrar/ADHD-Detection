import bcrypt
from database import get_connection

# ============================
# REGISTER USER

# ============================
def register_user(username, password, role="user"):
    conn = get_connection()
    cursor = conn.cursor()

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    cursor.execute(
        "INSERT INTO users (username, password, role) VALUES (%s,%s,%s)",
        (username, hashed.decode(), role)
    )

    conn.commit()
    conn.close()


# ============================
# LOGIN USER
# ============================
def login_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, password, role FROM users WHERE username=%s",
        (username,)
    )

    user = cursor.fetchone()
    conn.close()

    if user:
        user_id = user[0]
        hashed_pw = user[1]
        role = user[2]

        if bcrypt.checkpw(password.encode(), hashed_pw.encode()):
            return {"id": user_id, "role": role}

    return None