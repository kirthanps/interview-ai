# 90-second InterviewAI walkthrough

Use the [fictional resume](mira-vale-resume.pdf) and [job brief](job-description.md). Generate five questions and complete the interview once before recording. Use a second tab for scoring and the downloaded PDF if provider latency would interrupt the narration.

| Time | Screen | What to show |
| --- | --- | --- |
| 0–10 s | Landing | Introduce InterviewAI as a mock interview coach that adapts to a resume and a target role. |
| 10–25 s | Setup | Upload Mira Vale’s PDF. Paste the job description and company context into their respective fields. Select five questions and Mixed focus. |
| 25–40 s | Preview | Point out the extracted skills and role, then open **View generated questions**. |
| 40–60 s | Interview | Open a hint, enter a substantive answer, and submit it. Show the progress update. |
| 60–78 s | Results | Explain the overall and per-question scores, then open one question’s feedback and point out an improvement suggestion. |
| 78–90 s | Report | Download and open the PDF. Show the role, scores, and one feedback item. |

For a question about duplicate webhooks, try this answer: “I would store the provider’s event ID under a unique constraint and apply the event and status update in the same transaction. Retries would return success once the event is committed. I’d test duplicate delivery and inspect retry rates to catch regressions.” Adapt the answer to the generated question.

Record the browser and PDF viewer using your existing screen recorder. Trim time spent waiting for generation and avoid showing the terminal or configuration files.
