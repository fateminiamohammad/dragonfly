import { Injectable, Logger, OnModuleInit, UnauthorizedException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { InjectRepository } from '@nestjs/typeorm';
import * as bcrypt from 'bcryptjs';
import { Repository } from 'typeorm';
import { config } from '../config';
import { User } from '../entities/user.entity';

@Injectable()
export class AuthService implements OnModuleInit {
  private readonly log = new Logger(AuthService.name);
  // compared against when an email is unknown, so response time doesn't reveal which emails exist
  private readonly dummyHash = bcrypt.hashSync('dragonfly-timing-guard', 12);

  constructor(
    @InjectRepository(User) private readonly users: Repository<User>,
    private readonly jwt: JwtService,
  ) {}

  /** Creates the first admin from ADMIN_EMAIL / ADMIN_PASSWORD when there are no users yet. Idempotent. */
  async onModuleInit() {
    if ((await this.users.count()) > 0) return;
    const { email, password } = config.admin;
    if (!email || !password) {
      this.log.warn('no users and no ADMIN_EMAIL/ADMIN_PASSWORD set: nobody can log in yet');
      return;
    }
    await this.users.save(this.users.create({ email: email.toLowerCase(), passwordHash: await bcrypt.hash(password, 12), role: 'admin' }));
    this.log.log(`created admin ${email}`);
  }

  async login(email: string, password: string) {
    const user = await this.users.findOne({ where: { email: email.toLowerCase() } });
    const ok = await bcrypt.compare(password, user?.passwordHash ?? this.dummyHash);
    if (!user || !ok) throw new UnauthorizedException('invalid email or password');
    const token = await this.jwt.signAsync({ sub: user.id, email: user.email, role: user.role });
    return { token, user: { id: user.id, email: user.email, role: user.role } };
  }

  async createUser(email: string, password: string, role: 'admin' | 'member') {
    const user = await this.users.save(
      this.users.create({ email: email.toLowerCase(), passwordHash: await bcrypt.hash(password, 12), role }),
    );
    return { id: user.id, email: user.email, role: user.role };
  }
}
