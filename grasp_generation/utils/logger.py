"""
Last modified date: 2022.03.11
Author: mzhmxzh
Description: logger
"""

from torch.utils.tensorboard.writer import SummaryWriter


class Logger:
    def __init__(self, log_dir, thres_fc=0.3, thres_dis=0.005, thres_pen=0.02):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.thres_fc = thres_fc
        self.thres_dis = thres_dis
        self.thres_pen = thres_pen

    def log(self, energy, E_fc, E_dis, E_pen, E_spen, E_joints, step, show=False):
        success_fc = E_fc < self.thres_fc
        success_dis = E_dis < self.thres_dis
        success_pen = E_pen < self.thres_pen
        success = success_fc * success_dis * success_pen
        self.writer.add_scalar('Energy/energy', energy.mean(), step)
        self.writer.add_scalar('Energy/fc', E_fc.mean(), step)
        self.writer.add_scalar('Energy/dis', E_dis.mean(), step)
        self.writer.add_scalar('Energy/pen', E_pen.mean(), step)

        self.writer.add_scalar('Success/success', success.float().mean(), step)
        self.writer.add_scalar('Success/fc', success_fc.float().mean(), step)
        self.writer.add_scalar('Success/dis', success_dis.float().mean(), step)
        self.writer.add_scalar('Success/pen', success_pen.float().mean(), step)

        if show:
            print(f'Step %d energy: %f  fc: %f  dis: %f  pen: %f  spen: %f  joints: %f' % (step, energy.mean(), E_fc.mean(), E_dis.mean(), E_pen.mean(), E_spen.mean(), E_joints.mean()))
            print(f'success: %f  fc: %f  dis: %f  pen: %f' % (success.float().mean(), success_fc.float().mean(), success_dis.float().mean(), success_pen.float().mean()))
