import { Column, CreateDateColumn, Entity, Index, JoinColumn, ManyToOne, PrimaryGeneratedColumn } from 'typeorm';
import { User } from './user.entity';

/** Only the SHA-256 digest of a key is stored; the plaintext is shown once, at creation. */
@Entity('api_keys')
export class ApiKey {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column()
  name: string;

  /** First characters of the key, so people can tell their keys apart ("df_a1b2c3"). */
  @Column()
  prefix: string;

  @Index({ unique: true })
  @Column()
  digest: string;

  @ManyToOne(() => User, (u) => u.keys, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'user_id' })
  user: User;

  @Column({ name: 'user_id' })
  userId: string;

  @CreateDateColumn({ name: 'created_at', type: 'timestamptz' })
  createdAt: Date;

  @Column({ name: 'revoked_at', type: 'timestamptz', nullable: true })
  revokedAt: Date | null;
}
