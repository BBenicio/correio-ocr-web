import { DateTime } from 'luxon'
import { BaseModel, column, hasOne, HasOne } from '@ioc:Adonis/Lucid/Orm'
import File from 'App/Models/File'

export default class Job extends BaseModel {
  @column({ isPrimary: true })
  public id: number

  @hasOne(() => File)
  public file: HasOne<typeof File>

  @column()
  public finished: boolean

  @column.dateTime({ autoCreate: true })
  public createdAt: DateTime

  @column.dateTime({ autoCreate: true, autoUpdate: true })
  public updatedAt: DateTime
}
