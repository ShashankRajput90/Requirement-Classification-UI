import re

KEYWORDS = {
    "security": {
        "secure": "Indicates the system must protect data from unauthorized access.",
        "encrypt": "Encryption ensures sensitive data is protected.",
        "authentication": "Authentication verifies user identity.",
        "password": "Passwords are security credentials.",
        "login": "Login functionality relates to user authentication.",
        "ssl": "SSL provides encrypted communication over networks.",
        "https": "HTTPS ensures secure data transmission.",
        "encryption": "Encryption is the process of encoding data to prevent unauthorized access.",
        "encrypted": "Data that has been transformed to prevent unauthorized access.",
        "hash": "Hashing is a method of protecting data by transforming it into a fixed-length string.",
        "salt": "Salt is random data added to passwords before hashing to enhance security."    
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
    },
    "scalability": {
        "scalable": "System can handle growth in users or data.",
        "scale": "Ability to increase capacity when needed.",
        "load balancing": "Distributes traffic across servers.",
        "horizontal scaling": "Adding more machines to handle load.",
        "vertical scaling": "Increasing power of existing machines."
        },
    "maintainability": {
    "maintain": "Ease of maintaining the system.",
    "modular": "System is divided into independent modules.",
    "readable": "Code should be easy to understand.",
    "testable": "System should support testing.",
    "documentation": "Proper documentation improves maintainability."
     },
    "compatibility": {
        "compatible": "System can work with other systems or platforms.",
        "interoperable": "Ability to exchange and use information across systems.",
        "cross-platform": "Works on multiple operating systems.",
        "integration": "Ability to integrate with other software."
    },
    "accessibility": {
        "accessibility": "System can be used by people with disabilities.",
        "screen reader": "Supports screen readers for visually impaired users.",
        "keyboard navigation": "Allows navigation using a keyboard.",
        "color contrast": "Ensures sufficient contrast for readability.",
        "accessible": "Designs that accommodate all users, including those with disabilities."
    },
    
    "testability": {
            "testable": "System can be easily tested.",
            "unit test": "Supports unit testing for individual components.",
            "integration test": "Supports testing of combined components.",
            "automated testing": "Allows for automated test execution."
    },
    "modularity": {
                "modular": "System is divided into independent modules.",
                "component": "System is built with reusable components.",
                "separation of concerns": "Different concerns are handled by different modules."
    },
    "compliance": {
    "gdpr": "Ensures user data privacy compliance.",
    "regulation": "System follows legal standards.",
    "policy": "Adheres to organizational rules.",
    "audit": "Tracks system activity for compliance."
    },
    "interoperability": {
    "api": "Allows systems to communicate.",
    "integration": "Combines multiple systems.",
    "third-party": "Supports external services.",
    "data exchange": "Sharing data between systems."
  },
    "availability": {
    "uptime": "Measures system availability.",
    "redundancy": "Provides backup components.",
    "failover": "Automatically switches to backup systems.",
    "disaster recovery": "Plans for recovery from major failures."
    },
    "extensibility": {
    "extend": "System can be expanded with new features.",
    "plugin": "Supports additional modules.",
    "customization": "Allows user-defined changes.",
    "flexible": "Adapts to changing requirements."
   },
    "portability": {
    "portable": "System can be easily moved to different environments.",
    "containerization": "Using containers for consistent deployment.",
    "cloud": "Ability to run in cloud environments.",
    "cross-platform": "Supports multiple operating systems.",
    "migration": "Ability to move system across environments."
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