import { Body, Controller, Delete, Get, Param, ParseUUIDPipe, Post } from '@nestjs/common';
import { IsString, Length } from 'class-validator';
import { AuthUser, CurrentUser } from '../auth/auth.guard';
import { KeysService } from './keys.service';

class CreateKeyDto {
  @IsString()
  @Length(1, 80)
  name: string;
}

@Controller('keys')
export class KeysController {
  constructor(private readonly keys: KeysService) {}

  @Get()
  list(@CurrentUser() user: AuthUser) {
    return this.keys.list(user.id);
  }

  @Post()
  create(@CurrentUser() user: AuthUser, @Body() dto: CreateKeyDto) {
    return this.keys.create(user.id, dto.name);
  }

  @Delete(':id')
  revoke(@CurrentUser() user: AuthUser, @Param('id', ParseUUIDPipe) id: string) {
    return this.keys.revoke(user.id, id);
  }
}
