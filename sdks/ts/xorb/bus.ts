import { connect, ConsumerConfig, Msg } from 'nats';

export class NatsBus {
  private constructor() {}

  static async publish(subject: string, data: Uint8Array) {
    const nc = await connect();
    await nc.publish(subject, data);
    await nc.flush();
    await nc.close();
  }

  static async subscribe(subject: string, handler: (err: Error | null, msg: Msg) => void) {
    const nc = await connect();
    const sub = nc.subscribe(subject, { flowControl: true, idleHeartbeat: 5000 });
    for await (const msg of sub) {
      handler(null, msg);
    }
    await nc.close();
  }

  static durableConsumerConfig(stream: string, durable: string): ConsumerConfig {
    return {
      durable_name: durable,
      ack_wait: 30000,
      max_ack_pending: 1024,
      flow_control: true,
      idle_heartbeat: 5000,
      deliver_subject: `durables.${stream}.${durable}`
    };
  }
}
