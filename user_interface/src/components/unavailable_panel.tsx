import { Icon } from './icons';

type PlannedTab = 'encode' | 'decode' | 'train';
const content = {
  encode: {
    title: 'A message, hidden in plain sight.',
    description:
      'Turn an image and a private message into an image you can share.',
    steps: [
      'Choose a cover image',
      'Add your message',
      'Save the encoded image',
    ],
    label: 'Image encoding is not available yet.',
  },
  decode: {
    title: 'Bring the message back.',
    description:
      'Recover a message from an image with a matching trained decoder.',
    steps: [
      'Choose an encoded image',
      'Use the matching decoder',
      'Recover your message',
    ],
    label: 'Image decoding is not available yet.',
  },
  train: {
    title: 'Build a model for your images.',
    description:
      'Train, compare, and save an encoder and decoder in one local workspace.',
    steps: [
      'Prepare your dataset',
      'Train and review progress',
      'Save your model pair',
    ],
    label: 'Model training is not available yet.',
  },
};

/** Explain upcoming workflows without suggesting that they can run. */
export function UnavailablePanel({ tab }: { tab: PlannedTab }) {
  const details = content[tab];
  return (
    <div className="workspace_content">
      <div className="section_heading">
        <span className="eyebrow">YOUR WORKSPACE</span>
        <h1>{details.title}</h1>
        <p>{details.description}</p>
      </div>
      <section className="unavailable_card" aria-label={details.label}>
        <div className="preview_art" aria-hidden="true">
          <div className="art_grid" />
          <div className="art_sheet art_back" />
          <div className="art_sheet art_front">
            <Icon name={tab} />
            <span>STEGO / LAB</span>
          </div>
          <span className="art_dot" />
        </div>
        <div className="unavailable_body">
          <span className="pill">COMING IN A LATER SPRINT</span>
          <h2>{details.label}</h2>
          <p>
            This foundation release has no trained models. You can explore the
            workspace and save defaults in Config.
          </p>
          <ol className="workflow_steps">
            {details.steps.map((step, index) => (
              <li key={step}>
                <span>{index + 1}</span>
                {step}
              </li>
            ))}
          </ol>
        </div>
      </section>
      <div className="note_row">
        <span aria-hidden="true">◌</span>
        <p>Model quality and message recovery have not been measured.</p>
      </div>
    </div>
  );
}
