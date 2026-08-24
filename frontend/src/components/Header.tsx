interface HeaderProps {
  children?: React.ReactNode;
}

/** Application title bar with product name and tagline. */
export default function Header({ children }: HeaderProps) {
  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-4 sm:px-6">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-slate-900">
            SchemaRAG
          </h1>
          <p className="text-sm text-slate-500">
            Natural-language questions, grounded Text-to-SQL, read-only
            PostgreSQL execution.
          </p>
        </div>
        {children}
      </div>
    </header>
  );
}
