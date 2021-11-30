"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const promises_1 = require("fs/promises");
const Hash_1 = __importDefault(global[Symbol.for('ioc.use')]("Adonis/Core/Hash"));
const File_1 = __importDefault(global[Symbol.for('ioc.use')]("App/Models/File"));
const Job_1 = __importDefault(global[Symbol.for('ioc.use')]("App/Models/Job"));
class SubmissionsController {
    async index(ctx) {
        return ctx.view.render('home');
    }
    async submit(ctx) {
        const file = ctx.request.file('file', { extnames: ['pdf'] });
        if (file === null || file.hasErrors) {
            return ctx.response.badRequest('Por favor, envie um arquivo PDF válido');
        }
        const fileContents = await promises_1.readFile(file.tmpPath);
        const fileHash = await Hash_1.default.make(fileContents.toString('hex'));
        const foundFile = await File_1.default.findBy('file_hash', fileHash);
        if (foundFile !== null) {
            return ctx.response.redirect(`/document/${file.clientName}`);
        }
        const fileSub = await File_1.default.create({
            fileName: file.clientName,
            fileHash: fileHash,
            fileContent: fileContents,
            submitterIp: ctx.request.ip(),
        });
        const job = await Job_1.default.create({
            finished: false,
        });
        await job.related('file').save(fileSub);
        return ctx.response.redirect(`/processingSubmission?jobId=${job.id}`);
    }
    async checkCompletion(ctx) {
        const jobId = ctx.request.input('jobId');
        if (jobId === null) {
            return ctx.response.badRequest('Por favor, informe qual job quer verificar o status.');
        }
        const job = await Job_1.default.find(jobId);
        if (job === null) {
            return ctx.response.badRequest(`Não foi possível encontrar um job com o id ${jobId}`);
        }
        if (job.finished) {
            return ctx.response.redirect(`/document/${job.file.fileName}`);
        }
        return ctx.view.render('wait');
    }
}
exports.default = SubmissionsController;
//# sourceMappingURL=SubmissionsController.js.map