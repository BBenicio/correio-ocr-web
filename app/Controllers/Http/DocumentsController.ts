import { HttpContextContract } from '@ioc:Adonis/Core/HttpContext'
import Output from 'App/Models/Output'

export default class DocumentsController {
  public async viewPage(ctx: HttpContextContract) {
    const fileName = ctx.request.param('filename');
    if (fileName === null) {
      return ctx.response.notFound()
    }

    const pageNumber = Number.parseInt(ctx.request.param('pageNumber', '1'))
    const out = await Output.query().where('file_name', fileName).andWhere('page_number', pageNumber).first()
    if (out === null) {
      return ctx.response.notFound()
    }
    
    const edition = await Output.query().where('file_name', fileName)
    if (edition === null) {
      return ctx.response.notFound()
    }

    const pageCount = edition.map((value) => value.pageNumber).reduce((prev, curr) => Math.max(prev, curr))

    const previousPage = pageNumber > 0 ? pageNumber - 1 : null
    const nextPage = pageNumber < pageCount ? pageNumber + 1 : null

    return ctx.view.render('document', { fileName, pageNumber, previousPage, nextPage })
  }
}
