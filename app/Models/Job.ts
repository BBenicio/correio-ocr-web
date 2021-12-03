import { DateTime } from 'luxon'
import { BaseModel, BelongsTo, belongsTo, column } from '@ioc:Adonis/Lucid/Orm'
import File from 'App/Models/File'

export default class Job extends BaseModel {
  @column({ isPrimary: true })
  public id: number

  @column()
  public fileId: number

  @belongsTo(() => File)
  public file: BelongsTo<typeof File>

  @column()
  public pageCount: number

  @column()
  public outputPath: string

  @column()
  public finished: boolean

  @column.dateTime({ autoCreate: true })
  public createdAt: DateTime

  @column.dateTime({ autoCreate: true, autoUpdate: true })
  public updatedAt: DateTime
}
