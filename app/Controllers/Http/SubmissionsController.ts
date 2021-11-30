import { readFile } from 'fs/promises'
import { HttpContextContract } from '@ioc:Adonis/Core/HttpContext'
import Hash from '@ioc:Adonis/Core/Hash'
import File from 'App/Models/File'
import Job from 'App/Models/Job'

export default class SubmissionsController {
  public async index(ctx: HttpContextContract) {
    return ctx.view.render('home')
  }

  public async submit(ctx: HttpContextContract) {
    const file = ctx.request.file('file', { extnames: ['pdf'] })
    if (file === null || file.hasErrors) {
      return ctx.response.badRequest('Por favor, envie um arquivo PDF válido')
    }

    const fileContents = await readFile(file.tmpPath!)
    const fileHash = await Hash.make(fileContents.toString('hex'))
    const foundFile = await File.findBy('file_hash', fileHash)
    if (foundFile !== null) {
      return ctx.response.redirect(`/document/${file.clientName}`)
    }
    
    const fileSub = await File.create({
      fileName: file.clientName,
      fileHash: fileHash,
      fileContent: fileContents,
      submitterIp: ctx.request.ip(),
    })

    const job = await Job.create({
      finished: false,
    })
    await job.related('file').save(fileSub)

    return ctx.response.redirect(`/processingSubmission?jobId=${job.id}`)
  }
  
  public async checkCompletion(ctx: HttpContextContract) {
    const jobId = ctx.request.input('jobId')
    if (jobId === null) {
      return ctx.response.badRequest('Por favor, informe qual job quer verificar o status.')
    }
    
    const job = await Job.find(jobId)
    if (job === null) {
      return ctx.response.badRequest(`Não foi possível encontrar um job com o id ${jobId}`)
    }
    
    // check if output file exists
    if (job.finished) {
      return ctx.response.redirect(`/document/${job.file.fileName}`)
    }
    
    return ctx.view.render('wait')
  }
}
