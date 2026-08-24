export const EXAMPLE_QUESTIONS: string[] = [
  "Which department has the highest average marks?",
  "Show the top 5 students by marks.",
  "Which students have attendance below 75%?",
  "How many students are there?",
  "How many courses are offered by each department?",
];

interface ExampleQuestionsProps {
  onSelect: (question: string) => void;
  disabled?: boolean;
}

/** Clickable example questions derived from the seeded college schema. */
export default function ExampleQuestions({ onSelect, disabled }: ExampleQuestionsProps) {
  return (
    <section aria-label="Example questions" className="space-y-2">
      <h2 className="text-sm font-medium text-slate-600">Example questions</h2>
      <div className="flex flex-wrap gap-2">
        {EXAMPLE_QUESTIONS.map((q) => (
          <button
            key={q}
            type="button"
            disabled={disabled}
            onClick={() => onSelect(q)}
            className="rounded-full border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700 transition-colors hover:border-indigo-400 hover:bg-indigo-50 hover:text-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {q}
          </button>
        ))}
      </div>
    </section>
  );
}
