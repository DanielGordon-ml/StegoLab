interface IconProperties {
  name: 'encode' | 'decode' | 'train' | 'config' | 'brand';
}

/** Local line artwork; the workspace needs no network fonts or icons. */
export function Icon({ name }: IconProperties) {
  const paths = {
    encode: (
      <>
        <rect x="3" y="3" width="18" height="18" rx="4" />
        <path d="m4 16 5-5 4 4 3-3 5 5M15 7h.01" />
      </>
    ),
    decode: (
      <>
        <rect x="4" y="4" width="16" height="16" rx="4" />
        <path d="M8 9h8M8 12h8M8 15h5" />
      </>
    ),
    train: (
      <>
        <path d="M4 19V5m0 14h16M8 15l4-5 4 2 4-7" />
      </>
    ),
    config: (
      <>
        <path d="M4 7h16M4 17h16" />
        <circle cx="9" cy="7" r="3" />
        <circle cx="15" cy="17" r="3" />
      </>
    ),
    brand: (
      <>
        <path d="m12 2 10 6-10 6L2 8l10-6Zm-9 11 9 5 9-5M3 18l9 5 9-5" />
      </>
    ),
  };
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {paths[name]}
    </svg>
  );
}
