import { HttpContextContract } from '@ioc:Adonis/Core/HttpContext'
import Output from 'App/Models/Output'
import File from 'App/Models/File';
import Job from 'App/Models/Job';

export default class DocumentsController {
  public async viewPage(ctx: HttpContextContract) {
    const fileId = ctx.request.param('fileId');
    if (fileId === null) {
      return ctx.response.notFound()
    }
    const fileSub = await File.find(fileId)
    if (fileSub === null) {
      return ctx.response.notFound()
    }
    const pageNumber = Number.parseInt(ctx.request.param('pageNumber', '1'))

    const job = await Job.findBy('fileId', fileId)
    if (job === null) {
      return ctx.response.notFound()
    }

    const out = await Output.query().where('file_id', fileId).andWhere('page_number', pageNumber).first()
    if (out === null) {
      return ctx.response.notFound()
    }

    const pageCount = job.pageCount

    const previousPage = pageNumber > 0 ? pageNumber - 1 : null
    const nextPage = pageNumber < pageCount ? pageNumber + 1 : null

    return ctx.view.render('document', { fileName: fileSub.fileName, pageNumber, previousPage, nextPage, image: `/image/${out.id}`, text: `/text/${out.id}` })
  }

  public async getText(ctx: HttpContextContract) {
    const outId = ctx.request.param('outId');
    if (outId === null) {
      return ctx.response.notFound()
    }
    
    const out = await Output.find(outId)
    if (out === null) {
      return ctx.response.notFound()
    }
    await out.load('file')
    
    ctx.response.header('content-disposition', `inline; filename="${out.file.fileHash}.txt"`)
    ctx.response.type('.txt')
    return out.text
  }

  public async getImage(ctx: HttpContextContract) {
    const outId = ctx.request.param('outId');
    if (outId === null) {
      return ctx.response.notFound()
    }
    
    const out = await Output.find(outId)
    if (out === null) {
      return ctx.response.notFound()
    }
    await out.load('file')

    ctx.response.header('content-disposition', `inline; filename="${out.file.fileHash}.png"`)
    ctx.response.type('.png')
    return out.pageImage
  }
}
