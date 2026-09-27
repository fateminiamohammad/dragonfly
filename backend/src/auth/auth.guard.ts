import {
  CanActivate,
  ExecutionContext,
  ForbiddenException,
  Injectable,
  SetMetadata,
  UnauthorizedException,
  createParamDecorator,
} from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { JwtService } from '@nestjs/jwt';
import type { Role } from '../entities/user.entity';

export interface AuthUser {
  id: string;
  email: string;
  role: Role;
}

const PUBLIC = 'public';
const ROLES = 'roles';
/** Route needs no login (health, login). */
export const Public = () => SetMetadata(PUBLIC, true);
/** Route needs one of these roles. */
export const Roles = (...roles: Role[]) => SetMetadata(ROLES, roles);
export const CurrentUser = createParamDecorator((_: unknown, ctx: ExecutionContext): AuthUser => {
  return ctx.switchToHttp().getRequest().user;
});

/** Global guard: every route needs a valid JWT unless marked @Public(). */
@Injectable()
export class JwtAuthGuard implements CanActivate {
  constructor(
    private readonly jwt: JwtService,
    private readonly reflector: Reflector,
  ) {}

  async canActivate(ctx: ExecutionContext): Promise<boolean> {
    const targets = [ctx.getHandler(), ctx.getClass()];
    if (this.reflector.getAllAndOverride<boolean>(PUBLIC, targets)) return true;
    const req = ctx.switchToHttp().getRequest();
    const header: string = req.headers.authorization ?? '';
    if (!header.startsWith('Bearer ')) throw new UnauthorizedException();
    try {
      const payload = await this.jwt.verifyAsync(header.slice(7));
      req.user = { id: payload.sub, email: payload.email, role: payload.role } satisfies AuthUser;
    } catch {
      throw new UnauthorizedException();
    }
    const roles = this.reflector.getAllAndOverride<Role[] | undefined>(ROLES, targets);
    if (roles && !roles.includes(req.user.role)) throw new ForbiddenException();
    return true;
  }
}
