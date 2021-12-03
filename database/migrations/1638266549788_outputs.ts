import BaseSchema from '@ioc:Adonis/Lucid/Schema'

export default class Outputs extends BaseSchema {
  protected tableName = 'outputs'

  public async up () {
    this.schema.createTable(this.tableName, (table) => {
      table.increments('id')
      table.integer('file_id').references('files.id').onDelete('CASCADE')
      table.integer('page_number')
      table.binary('page_image')
      table.text('text')
      
      /**
       * Uses timestamptz for PostgreSQL and DATETIME2 for MSSQL
       */
      table.timestamp('created_at', { useTz: true })
      table.timestamp('updated_at', { useTz: true })
    })
  }

  public async down () {
    this.schema.dropTable(this.tableName)
  }
}
