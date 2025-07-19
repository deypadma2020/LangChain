from langchain.text_splitter import CharacterTextSplitter

text = """
Situation:
During a cross-functional project at Klizo Solutions, 
we had to integrate financial, sales, and customer 
support data to build a unified reporting system. 
The team included analysts, developers, 
and QA members from different departments, 
and early on, communication gaps were slowing down progress.

Task:
As one of the core analysts, 
I noticed that unclear requirements and 
siloed work were leading to redundant efforts and delays. 
I wanted to boost collaboration and create a sense of shared ownership.

Action:
I proposed setting up short daily sync-ups with all team members 
and created a shared workspace where everyone could update their progress, 
dependencies, and blockers. I also organized a kickoff meeting to align everyone on the business goals, 
making sure each member saw how their work contributed to the larger outcome.

Additionally, I broke down complex tasks into smaller, 
well-defined parts so that each team member could contribute effectively 
and see quick wins, which helped boost morale.

Result:
The better communication flow and clear responsibilities helped us work more efficiently 
and reduce delivery time by 20%. The team environment improved significantly, 
and several members appreciated the structured yet collaborative approach. 
The final reporting system was adopted by senior stakeholders and used for monthly executive reviews.
"""

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_text(text)

print(result)