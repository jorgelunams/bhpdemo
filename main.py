import asyncio
import os
import dotenv

# Assuming your classify_question function is defined here
from OpenAIKernel import classify_question

# Load environment variables from .env file
dotenv.load_dotenv()

async def main():
    # Replace 'your_question_here' with your actual question
    result = await classify_question('Give me all stats about WAIO and Coal and add total production for WAIO and Coal?')
    print(result)

# Run the main function in an event loop
if __name__ == '__main__':
    asyncio.run(main())