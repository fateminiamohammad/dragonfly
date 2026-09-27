import { Body, Controller, Get, Post } from '@nestjs/common';
import { Throttle } from '@nestjs/throttler';
import { IsEmail, IsIn, IsString, MinLength } from 'class-validator';
import { AuthUser, CurrentUser, Public, Roles } from './auth.guard';
import { AuthService } from './auth.service';

class LoginDto {
  @IsEmail()
  email: string;

  @IsString()
  password: string;
}

class CreateUserDto {
  @IsEmail()
  email: string;

  @IsString()
  @MinLength(12)
  password: string;

  @IsIn(['admin', 'member'])
  role: 'admin' | 'member';
}

@Controller()
export class AuthController {
  constructor(private readonly auth: AuthService) {}

  @Public()
  @Throttle({ default: { limit: 10, ttl: 60_000 } })
  @Post('auth/login')
  login(@Body() dto: LoginDto) {
    return this.auth.login(dto.email, dto.password);
  }

  @Get('me')
  me(@CurrentUser() user: AuthUser) {
    return user;
  }

  @Roles('admin')
  @Post('users')
  createUser(@Body() dto: CreateUserDto) {
    return this.auth.createUser(dto.email, dto.password, dto.role);
  }
}
