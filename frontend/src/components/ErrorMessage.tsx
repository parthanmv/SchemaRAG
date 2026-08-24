interface ErrorMessageProps {
  title: string;
  message: string;
}

/** Accessible error alert. Shows only backend-provided, user-safe messages. */
export default function ErrorMessage({ title, message }: ErrorMessageProps) {
  return (
    <div
      role="alert"
      data-testid="error-message"
      className="rounded-lg border border-red-200 bg-red-50 px-4 py-3"
    >
      <p className="text-sm font-semibold text-red-800">{title}</p>
      <p className="mt-0.5 break-words text-sm text-red-700">{message}</p>
    </div>
  );
}
