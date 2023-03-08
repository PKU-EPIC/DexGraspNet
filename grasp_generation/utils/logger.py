"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: Class Logger for tensorboard
"""

from torch.utils.tensorboard.writer import SummaryWriter


class Logger:
    def __init__(self, log_dir, thres_fc=0.3, thres_dis=0.005, thres_pen=0.02):
        """
        Create a Logger to log tensorboard scalars
        
        Parameters
        ----------
        log_dir: str
            directory for logs
        thres_fc: float
            E_fc threshold for success estimation
        thres_dis: float
            E_dis threshold for success estimation
        thres_pen: float
            E_pen threshold for data filtering
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        self.thres_fc = thres_fc
        self.thres_dis = thres_dis
        self.thres_pen = thres_pen

    def log(self, energy, E_fc, E_dis, E_pen, E_spen, E_joints, step, show=False):
        """
        Log energy terms and estimate success rate using energy thresholds
        
        Parameters
        ----------
        energy: torch.Tensor
            weighted sum of all terms
        E_fc: torch.Tensor
        E_dis: torch.Tensor
        E_pen: torch.Tensor
        E_spen: torch.Tensor
        E_joints: torch.Tensor
        step: int
            current iteration of optimization
        show: bool
            whether to print current energy terms to console
        """
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
