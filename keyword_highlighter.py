import re

KEYWORDS = {
    "security": {
        "secure": "Indicates the system must protect data from unauthorized access.",
        "encrypt": "Encryption ensures sensitive data is protected.",
        "authentication": "Authentication verifies user identity.",
        "password": "Passwords are security credentials.",
        "login": "Login functionality relates to user authentication.",
        "ssl": "SSL provides encrypted communication over networks.",
        "https": "HTTPS ensures secure data transmission."
    },

    "performance": {
        "fast": "Fast systems reduce waiting time.",
        "latency": "Latency measures delay in system response.",
        "throughput": "Throughput refers to system processing capacity.",
        "response time": "Response time measures how quickly the system reacts.",
        "efficient": "Efficiency indicates optimized resource usage."
    },

    "usability": {
        "easy": "Ease of use improves user experience.",
        "user-friendly": "User-friendly interfaces improve usability.",
        "intuitive": "Intuitive systems require minimal learning.",
        "accessible": "Accessibility ensures usability for all users."
    },

    "reliability": {
        "availability": "Availability ensures the system remains operational.",
        "backup": "Backups allow data recovery in case of failure.",
        "recover": "Recovery mechanisms restore system functionality.",
        "uptime": "Uptime measures system reliability."
    }
}


def highlight_keywords(text):
    highlighted = text

    for category, words in KEYWORDS.items():
        for word, explanation in words.items():
            pattern = re.compile(rf"\b({word})\b", re.IGNORECASE)

            highlighted = pattern.sub(
                rf'<span class="keyword-highlight" data-explanation="{explanation}">\1</span>',
                highlighted
            )

    return highlighted