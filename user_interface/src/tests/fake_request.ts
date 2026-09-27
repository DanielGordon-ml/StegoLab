/** Stand in for the browser request so each outcome can be driven by hand. */
export class FakeRequest {
  static current: FakeRequest;
  /** Every request sent so far, in order, since the last reset. */
  static sent_requests: FakeRequest[] = [];
  private static unclaimed: FakeRequest[] = [];
  private static waiters: ((request: FakeRequest) => void)[] = [];
  status = 0;
  responseText = '';
  timeout = 0;
  responseType = '';
  headers: Record<string, string> = {};
  upload = { onprogress: null as ((event: ProgressEvent) => void) | null };
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  ontimeout: (() => void) | null = null;
  onabort: (() => void) | null = null;
  sent: unknown = null;
  url = '';
  method = '';
  aborted = false;
  constructor() {
    FakeRequest.current = this;
  }
  open(method: string, url: string) {
    this.method = method;
    this.url = url;
  }
  setRequestHeader(name: string, value: string) {
    this.headers[name] = value;
  }
  send(body: unknown) {
    this.sent = body;
    FakeRequest.sent_requests.push(this);
    const waiter = FakeRequest.waiters.shift();
    if (waiter) waiter(this);
    else FakeRequest.unclaimed.push(this);
  }
  abort() {
    this.aborted = true;
    this.onabort?.();
  }
  /** Answer the way a server would, then fire the load handler. */
  respond(status: number, body: unknown) {
    this.status = status;
    this.responseText = typeof body === 'string' ? body : JSON.stringify(body);
    this.onload?.();
  }
  /** Resolve with the next request the code under test sends. */
  static next_send(): Promise<FakeRequest> {
    const ready = FakeRequest.unclaimed.shift();
    if (ready) return Promise.resolve(ready);
    return new Promise((resolve) => FakeRequest.waiters.push(resolve));
  }
  /** Forget earlier requests so one test cannot see another's traffic. */
  static reset() {
    FakeRequest.sent_requests = [];
    FakeRequest.unclaimed = [];
    FakeRequest.waiters = [];
  }
}
