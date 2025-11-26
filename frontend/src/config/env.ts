export const withDefault = (value: string | undefined, fallback: string) => {
  const trimmed = value?.trim();
  return trimmed && trimmed.length > 0 ? trimmed : fallback;
};


