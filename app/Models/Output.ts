import { DateTime } from 'luxon'
import { BaseModel, belongsTo, BelongsTo, column } from '@ioc:Adonis/Lucid/Orm'
import File from 'App/Models/File'

export default class Output extends BaseModel {
  @column({ isPrimary: true })
  public id: number

  @column()
  public fileId: number

  @belongsTo(() => File)
  public file: BelongsTo<typeof File>

  @column()
  public pageNumber: number

  @column()
  public pageImage: Buffer

  @column()
  public text: string

  @column.dateTime({ autoCreate: true })
  public createdAt: DateTime

  @column.dateTime({ autoCreate: true, autoUpdate: true })
  public updatedAt: DateTime
}
