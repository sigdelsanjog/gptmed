import sys
from .startproject import startproject

def main():
    if len(sys.argv) < 3 or sys.argv[1] != "startproject":
        print("Usage: gptmed startproject <projectname> [--qna|--conversational]")
        sys.exit(1)
    
    project_name = sys.argv[2]
    project_type = None
    
    # Check for flags
    if len(sys.argv) > 3:
        flag = sys.argv[3]
        if flag == "--qna":
            project_type = "qna"
        elif flag == "--conversational":
            project_type = "conversational"
        else:
            print(f"Invalid flag: {flag}")
            print("Usage: gptmed startproject <projectname> [--qna|--conversational]")
            sys.exit(1)
    
    startproject(project_name, project_type)
